from __future__ import annotations

import random
from typing import Any, Mapping, Optional

import numpy as np
import torch
from pathlib import Path

from peft import PeftModel

from .utils import save_config
from .types import PathInput


def get_rng_state() -> dict[str, Any]:
    """Capture all RNG states into a serializable dict.

    Returns keys: python_random, numpy, torch_cpu, torch_cuda (list of
    per-device states, or None if CUDA unavailable).
    Does NOT capture DataLoader generator state — that is replayed from
    seed on resume.
    """
    state = {
        "python_random": random.getstate(),
        "numpy": np.random.get_state(),
        "torch_cpu": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["torch_cuda"] = torch.cuda.get_rng_state_all()
    else:
        state["torch_cuda"] = None
    return state


def set_rng_state(state: Mapping[str, Any]) -> None:
    """Restore RNG states from a dict produced by get_rng_state.

    CUDA keys are skipped if CUDA is unavailable (CPU-only resume).
    """
    random.setstate(state["python_random"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch_cpu"])
    if state.get("torch_cuda") is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["torch_cuda"])


def save_checkpoint(
    transformer: Any,
    text_encoder: Any,
    step: int,
    output_dir: PathInput,
    global_step: int,
    micro_step: int,
    rng_state: Mapping[str, Any],
    transformer_opt: Any,
    qwen_opt: Any,
    config: Mapping[str, Any],
) -> Path:
    """Save full training state: LoRA weights, optimizer, counters, RNG, config.

    Directory structure:
      checkpoint-{step}/
        transformer_lora/      (if transformer is PeftModel)
        qwen_lora/             (if text_encoder is PeftModel)
        training_state.pt      (counters + RNG + optimizer states)
        config.yaml            (snapshot of training config)
    """
    ckpt_dir = Path(output_dir) / f"checkpoint-{step}"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    if isinstance(transformer, PeftModel):
        transformer.save_pretrained(ckpt_dir / "transformer_lora")

    if isinstance(text_encoder, PeftModel):
        text_encoder.save_pretrained(ckpt_dir / "qwen_lora")

    optimizer_states = {
        "transformer": transformer_opt.state_dict() if transformer_opt is not None else None,
        "qwen": qwen_opt.state_dict() if qwen_opt is not None else None,
    }

    training_state = {
        "global_step": global_step,
        "micro_step": micro_step,
        "rng_states": rng_state,
        "optimizer_states": optimizer_states,
    }
    torch.save(training_state, ckpt_dir / "training_state.pt")
    save_config(config, ckpt_dir / "config.yaml")

    print(f"Checkpoint saved: {ckpt_dir}")
    return ckpt_dir


def load_training_state(checkpoint_dir: PathInput) -> dict[str, Any]:
    """Load training_state.pt from a checkpoint directory.

    Returns dict with keys: global_step, micro_step, rng_states,
    optimizer_states (dict with 'transformer' and 'qwen' keys).

    Raises FileNotFoundError if training_state.pt is missing.
    """
    ckpt_dir = Path(checkpoint_dir)
    state_path = ckpt_dir / "training_state.pt"
    if not state_path.exists():
        raise FileNotFoundError(
            f"No training_state.pt found in {ckpt_dir}. "
            "This checkpoint was likely saved before resume support was added."
        )
    return torch.load(state_path, map_location="cpu", weights_only=False)
