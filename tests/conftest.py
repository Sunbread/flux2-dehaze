"""Shared fixtures, markers, and VRAM guard for dehaze_lora tests."""

import gc
import json
import os
import tempfile
from pathlib import Path

import pytest
import torch
from PIL import Image


# ---------------------------------------------------------------------------
# GPU / VRAM utilities
# ---------------------------------------------------------------------------

def get_free_vram_gb() -> float:
    """Return free GPU VRAM in GB, or inf if no CUDA device."""
    if not torch.cuda.is_available():
        return float("inf")
    free_bytes, _ = torch.cuda.mem_get_info()
    return free_bytes / (1024**3)


def _require_vram_gb(required: float):
    """Skip the current test if free VRAM is below *required* GB."""
    free = get_free_vram_gb()
    if free < required:
        pytest.skip(
            f"Need {required:.1f} GB free VRAM, only {free:.1f} GB available"
        )


def _require_cuda():
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")


def cleanup_gpu():
    """Run garbage collection + empty CUDA cache.

    Caller must dereference GPU tensors/models first (set to None or
    del), otherwise GC cannot collect them.
    """
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


# ---------------------------------------------------------------------------
# pytest markers
# ---------------------------------------------------------------------------

def pytest_configure(config):
    config.addinivalue_line("markers", "gpu: test requires CUDA GPU")
    config.addinivalue_line("markers", "slow: test is slow (needs real HF models)")


# ---------------------------------------------------------------------------
# temporary directories
# ---------------------------------------------------------------------------

@pytest.fixture
def tmp_dir():
    with tempfile.TemporaryDirectory() as d:
        yield Path(d)


# ---------------------------------------------------------------------------
# synthetic image fixtures (PIL, no GPU)
# ---------------------------------------------------------------------------

@pytest.fixture
def rgb_image_512():
    """A 512x512 RGB PIL image with deterministic content."""
    return Image.new("RGB", (512, 512), color=(128, 64, 32))


@pytest.fixture
def rgb_image_256():
    return Image.new("RGB", (256, 256), color=(200, 100, 50))


@pytest.fixture
def rgb_image_small():
    """64x64 RGB image for tiny model forward passes."""
    return Image.new("RGB", (64, 64), color=(100, 150, 200))


# ---------------------------------------------------------------------------
# synthetic metadata (JSONL)
# ---------------------------------------------------------------------------

@pytest.fixture
def dummy_metadata_jsonl(tmp_dir, rgb_image_512):
    """Write a small metadata JSONL file with 20 synthetic entries."""
    hazy_dir = tmp_dir / "hazy"
    gt_dir = tmp_dir / "gt"
    hazy_dir.mkdir()
    gt_dir.mkdir()

    lines = []
    for i in range(20):
        hazy_path = hazy_dir / f"hazy_{i:03d}.png"
        gt_path = gt_dir / f"gt_{i:03d}.png"
        rgb_image_512.save(hazy_path)
        rgb_image_512.save(gt_path)
        lines.append(json.dumps({
            "image": str(hazy_path),
            "gt": str(gt_path),
            "caption": "Dehaze test prompt",
        }))

    meta_path = tmp_dir / "metadata.jsonl"
    meta_path.write_text("\n".join(lines))
    return meta_path


# ---------------------------------------------------------------------------
# tiny LoRA-compatible module for injection tests
# ---------------------------------------------------------------------------

class TinyAttention(torch.nn.Module):
    """Minimal module with named linear layers matching LoRA target names."""
    def __init__(self, dim=64):
        super().__init__()
        self.to_q = torch.nn.Linear(dim, dim)
        self.to_k = torch.nn.Linear(dim, dim)
        self.to_v = torch.nn.Linear(dim, dim)
        self.to_out = torch.nn.Sequential(
            torch.nn.Linear(dim, dim),
            torch.nn.Dropout(0.0),
        )

    def forward(self, x):
        q = self.to_q(x)
        k = self.to_k(x)
        v = self.to_v(x)
        out = q * k * v  # dummy op
        return self.to_out(out)


class TinyQwenAttention(torch.nn.Module):
    """Minimal Qwen-like attention with q_proj/k_proj/v_proj/o_proj naming."""
    def __init__(self, dim=64):
        super().__init__()
        self.q_proj = torch.nn.Linear(dim, dim)
        self.k_proj = torch.nn.Linear(dim, dim)
        self.v_proj = torch.nn.Linear(dim, dim)
        self.o_proj = torch.nn.Linear(dim, dim)

    def forward(self, x):
        return self.o_proj(self.q_proj(x) + self.k_proj(x) + self.v_proj(x))


# ---------------------------------------------------------------------------
# shared model loader (used by GPU test files to avoid triplication)
# ---------------------------------------------------------------------------

MODEL_NAME = "black-forest-labs/FLUX.2-klein-base-9B"


def load_flux2_transformer():
    """Load Flux2 transformer from HF hub (bfloat16). Caller handles cleanup."""
    from diffusers import Flux2Transformer2DModel
    return Flux2Transformer2DModel.from_pretrained(
        MODEL_NAME, subfolder="transformer", torch_dtype=torch.bfloat16,
    )


# ---------------------------------------------------------------------------
# deterministic seed
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def reset_seed():
    """Reset RNG seeds before every test for reproducibility."""
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)
