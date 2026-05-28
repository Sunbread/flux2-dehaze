# Multi-GPU Training: 2x A100 40GB Support

Date: 2026-05-28

## Goal

Refactor training to run on 2x A100 40GB GPUs via manual model parallelism
(transformer+VAE on GPU0, Qwen3 on GPU1), plus optional gradient checkpointing.
Also supports single-GPU 80GB mode without checkpointing.

## Motivation

- 40GB GPU cannot hold all three base models (~35 GB) simultaneously
- FSDP/DDP incompatible with Muon optimizer (ShardedTensor breaks 2D constraint)
- accelerate has no built-in model parallelism support
- Drop accelerate entirely in favor of raw PyTorch — fewer abstractions fighting us

## Architecture

```
GPU 0 (cuda:0)                    GPU 1 (cuda:1)
┌──────────────────┐              ┌──────────────┐
│ VAE (frozen)     │              │ Qwen3 + LoRA │
│ Transformer+LoRA │              │ Tokenizer    │
│                  │              │              │
│ encode → latent  │              │ encode_prompt│
│   ↑              │  prompt_emb  │      ↓       │
│   │   .to(cuda:0)│←────────────│      │       │
│   │              │   text_ids   │      │       │
│   │              │←────────────│      │       │
│                  │      ↑       │              │
│  transformer fwd │      │ .grad │              │
│  loss.backward() │──────┘      │              │
│                  │ CopyBackward│              │
└──────────────────┘              └──────────────┘
```

- `prompt_embeds.to("cuda:0")` copies data but records `CopyBackwards` in autograd graph
- Backward gradients auto-route from GPU0 back to GPU1 Qwen3 LoRA params
- No manual gradient sync needed — PyTorch handles it

## Config Changes

```yaml
transformer_device: "cuda:0"    # new key
qwen_device: "cuda:1"           # new key
gradient_checkpointing: true    # new key, default false

# 2x40GB default
batch_size: 4
gradient_accumulation_steps: 8  # effective = 4×8 = 32

# For single 80GB: set batch_size=2, grad_accum=16, checkpointing=false
```

## File Changes

### `src/dehaze_lora/model.py` — Device-aware model loading

- `load_models()` gets `transformer_device` and `qwen_device` params
- VAE + transformer → `transformer_device`
- Qwen3 + tokenizer → `qwen_device`
- `encode_prompt()` stays on its natural device (Qwen3's device)

### `src/dehaze_lora/train.py` — Raw PyTorch training loop

**Remove:**
- All `accelerator.*` calls (~15 lines)
- `from accelerate import Accelerator`

**Add:**
```python
# Manual gradient accumulation
scaler = None  # bf16 doesn't need loss scaling
micro_step = 0
for batch in loader:
    with torch.cuda.amp.autocast(dtype=torch.bfloat16):
        # VAE on transformer_device (no_grad)
        # Qwen3 encode on qwen_device
        # .to(transformer_device) for prompt_embeds, text_ids
        # Transformer forward on transformer_device
        loss = flow_matching_loss(...) / grad_accum  # normalize for accumulation

    loss.backward()
    micro_step += 1

    if micro_step % grad_accum == 0:
        # Clip + step on respective devices
        torch.nn.utils.clip_grad_norm_(transformer.parameters(), t_clip)
        if qwen_opt:
            torch.nn.utils.clip_grad_norm_(text_encoder.parameters(), q_clip)
        transformer_opt.step(); transformer_opt.zero_grad()
        if qwen_opt: qwen_opt.step(); qwen_opt.zero_grad()
        global_step += 1
```

- `loss / grad_accum` because each backward adds rather than averages
- `torch.nn.utils.clip_grad_norm_()` replaces `accelerator.clip_grad_norm_()`
- `torch.manual_seed()` replaces accelerator seed management (already present)

### `src/dehaze_lora/train.py` — Checkpoint save

- Remove `accelerator.unwrap_model()` calls — model is bare PyTorch now
- Direct `PeftModel.save_pretrained()`

### `scripts/train.sh`

```bash
export CUDA_VISIBLE_DEVICES=0,1
# No more accelerate launch — direct python
uv run python -m src.dehaze_lora.train --config configs/config.yaml
```

### `configs/config.yaml` — Two presets

```yaml
# 2x40GB (default)
transformer_device: "cuda:0"
qwen_device: "cuda:1"
gradient_checkpointing: true
batch_size: 4
gradient_accumulation_steps: 8

# Single 80GB override
# transformer_device: "cuda:0"
# qwen_device: "cuda:0"
# gradient_checkpointing: false
# batch_size: 2
# gradient_accumulation_steps: 16
```

## Gradient Checkpointing

- `transformer.enable_gradient_checkpointing()` when config says true
- Reduces transformer activation from ~16 GB (bs=4) to ~4 GB
- Cost: recompute FFN + norms on backward pass (FlashAttn already saves Q/K/V only)
- Compensated by larger micro-batch (bs=4 instead of bs=2) → fewer forward/backward pairs

## VRAM Budget (2x40GB, bs=4, checkpointing on)

| GPU | Component | VRAM |
|:---:|------|:---:|
| 0 | VAE | 0.2 GB |
| 0 | Transformer base | 18 GB |
| 0 | Transformer activations (ckpt, bs=4) | ~4 GB |
| 0 | LoRA params + opt + grads | ~0.5 GB |
| 0 | CUDA overhead | ~2 GB |
| 0 | **Total** | **~25 GB** |
| 1 | Qwen3 base | 17 GB |
| 1 | Qwen3 activations (bs=4) | ~4 GB |
| 1 | LoRA params + opt + grads | ~0.4 GB |
| 1 | CUDA overhead | ~2 GB |
| 1 | **Total** | **~23 GB** |

Comfortable headroom on both GPUs.

## Test Changes

### `tests/test_training_step.py`
- Replace `Accelerator` mock with raw PyTorch flow
- Test manual gradient accumulation counting
- Test cross-device autograd (simulate two devices with CPU/CUDA split)

### Unchanged tests
- All CPU tests pass without changes (no accelerate dependency)
- GPU tests keep accelerate where they use it for loading convenience
- `test_model_loading.py`, `test_vae.py`, `test_checkpoint.py` — no changes needed

## Error Handling

- `load_models()` validates devices: `torch.cuda.is_available()` before `.to("cuda:N")`
- Same-device fallback warning when `transformer_device == qwen_device`
- `clip_grad_norm_` on models without gradients (frozen) — no-op, not an error

## What Stays The Same

- `optimizer.py` — Muon device-agnostic
- `loss.py`, `dataset.py`, `preprocess.py`, `validate.py`, `utils.py`
- LoRA injection logic in `model.py`
- `pyproject.toml` dependencies (remove `accelerate` optional)

## Dependencies

- **Remove**: `accelerate` from `pyproject.toml` (no longer needed at all)
- All other deps unchanged
