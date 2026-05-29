# FLUX.2 Klein Dehaze LoRA Training

Fine-tuning Black Forest Labs' FLUX.2 Klein (9B) transformer via LoRA for image dehazing. The model takes a hazy RGB image, processes it as a reference image (patchified with index 10.0), and uses classifier-free guidance (CFG) to produce a clear output via the flow-matching diffusion process.

## Quick Reference

| What | How |
|------|-----|
| Run training | `bash scripts/train.sh` |
| Run tests (CPU only) | `uv run pytest tests/ -m "not gpu and not slow" -v` |
| Run all tests with GPU | `uv run pytest tests/ -v` |
| Preprocess data | `uv run python -m src.dehaze_lora.preprocess` |
| Validate checkpoint | `uv run python -c "from src.dehaze_lora.validate import validate; validate(...)"` |
| Config file | `configs/config.yaml` |
| Module path | `src/dehaze_lora/` (editable install via `pyproject.toml`) |
| Package manager | `uv` (see `uv.lock` for exact deps) |

## Architecture

```
src/dehaze_lora/
  __init__.py     # Package marker
  types.py        # TypedDicts (DatasetItem, ValidationImageOutput, etc.), TypeAlias (PathInput, LoraTarget)
  model.py        # Model loading, LoRA injection, VAE encode/decode, text encoding, patchify/unpatchify
  loss.py         # Flow-matching MSE loss: target = noise - clean_latent
  optimizer.py    # Muon optimizer wrapper (2D-param constraint, match_rms_adamw LR adjustment)
  train.py        # Training loop with raw PyTorch, gradient accumulation, checkpointing, wandb logging
  validate.py     # Inference with three-mode denoising, PSNR/SSIM evaluation
  dataset.py      # DehazeDataset (10% caption dropout for CFG), DehazeValDataset (tests only), DEHAZE_PROMPT
  preprocess.py   # RESIDE and NH-HAZE dataset preprocessing pipeline
  checkpoint.py   # Checkpoint save/load with RNG state capture and training resume
  utils.py        # YAML config load/save
```

### Data Flow (Training Step)

1. Hazy + GT images loaded from dataset (512x512 RGB)
2. VAE encode both: image -> 32ch latents -> patchify 2x2 -> 128ch -> BN normalize
3. Text encoding via Qwen3 chat template -> stack hidden states from layers (9,18,27) -> joint_dim 12288
4. Timestep sampled via logit-normal distribution: sigma = sigmoid(N(0,1)), shifted with shift=3.0
5. Noisy latent = (1-sigma)*GT_latent + sigma*noise (sigma shape: (B,), applied via view(B,1,1,1))
6. Patchify: noisy (index=0.0) + hazy ref (index=10.0) concatenated as image tokens
7. Transformer forward -> extract noisy token outputs -> unpatchify
8. Flow-matching loss: MSE(model_pred, noise - GT_latent)
9. Gradient accumulation via manual `loss / grad_accum`, separate optimizers per model

### Inference (Three-Mode Validation)

validate.py runs three denoising trajectories from the same noise seed and reference tokens:

**Shared setup**: `compute_empirical_mu(image_seq_len, num_inference_steps)` calibrates the scheduler shift for the given resolution and step count.

**Mode 1 — Conditional generation** (no CFG): prompt-only denoising (`v = transformer(z + ref, cond_prompt)`).

**Mode 2 — Unconditional dehaze anchor** (uncond): empty-prompt-only denoising (`v = transformer(z + ref, empty_prompt)`). This learns p(Y | I), not p(I | I) — caption dropout removes only text; the flow target remains GT clear image.

**Mode 3 — CFG (classifier-free guidance)**:
```
z = pure_noise
for t in scheduler.timesteps:
    v_cond   = transformer(z + ref, cond_prompt)
    v_uncond = transformer(z + ref, empty_prompt)
    v_cfg = v_uncond + guidance_scale * (v_cond - v_uncond)
    z = scheduler.step(v_cfg, t, z).prev_sample
result = vae_decode(denormalize -> unpatchify -> decode)
```

## Immutable Constants (Do Not Change)

These come from the actual FLUX.2 Klein model config and ComfyUI source. Changing any of them will break the model.

### Transformer Config (Flux2Transformer2DModel)
| Constant | Value | Why |
|----------|-------|-----|
| `num_attention_heads` | 32 | NOT 48 -- verified against HF config.json |
| `attention_head_dim` | 128 | per-head dimension |
| `joint_attention_dim` | 12288 | 3 x Qwen3-4096 hidden states concatenated |
| `num_layers` (double) | 8 | double-stream blocks |
| `num_single_layers` | 24 | single-stream blocks |
| `patch_size` | 1 | transformer-level (not VAE-level) |
| `guidance_embeds` | False | Klein base has no guidance embedding |

### VAE (AutoencoderKLFlux2)
| Constant | Value | Why |
|----------|-------|-----|
| `latent_channels` | 32 | 8x spatial downscale |
| `patch_size` | [2, 2] | VAE-level patchify -> 128ch, 16x downscale |
| BatchNorm | Yes | `(latent - bn_mean) / bn_std` after patchify |
| process_in/out | identity | No shift/scale |

### Position Encoding
| Constant | Value |
|----------|-------|
| `axes_dims_rope` | [32, 32, 32, 32] |
| `rope_theta` | 2000 |
| hidden_size | 4096 (128 x 32) |
| pe_dim | 128 (must equal sum of axes_dims) |

### Text Encoding
| Constant | Value |
|----------|-------|
| Qwen3 hidden state layers | (9, 18, 27) |
| Chat template | Qwen3 with `add_generation_prompt=True, enable_thinking=False` |
| Unconditional prompt | `""` (empty string) goes through same chat template -- NOT a raw empty string |

### Prompt

| Constant | Value |
|----------|-------|
| `DEHAZE_PROMPT` | Fixed 50+ word prompt in `dataset.py:13-20` — used for both training and validation |

### LoRA Target Modules
| Model | Modules |
|-------|---------|
| Transformer (Flux2) | `["to_q", "to_k", "to_v", "to_out.0", "to_qkv_mlp_proj"]` |
| Qwen3 text encoder | `["q_proj", "k_proj", "v_proj", "o_proj"]` |

### Position IDs Are NOT Vocabulary Tokens

`img_ids` and `txt_ids` are (T, H, W, L) 4D RoPE spatial coordinates. T/H/W are zero for text; L is 0..seq_len-1. T=0 for noisy image, T=10 for reference image. These feed into the RoPE position embedder and have nothing to do with vocabulary token IDs. Confusing these is catastrophic.

## Critical Gotchas (Previously Fixed Silent Bugs)

These bugs produced no errors but silently broke training. Tests now catch them. Do NOT reintroduce any of these patterns.

### 1. Gradient Accumulation Uses Manual `loss / grad_accum`
The loss MUST be divided by `gradient_accumulation_steps` before `backward()`. Without scaling, each micro-batch's gradients accumulate additively and effectively multiply the LR by `gradient_accumulation_steps`.

### 2. Qwen LoRA Needs Gradient Tracking in `encode_prompt`
When Qwen has LoRA (lora_target="qwen" or "both"), `encode_prompt` must NOT be inside `torch.no_grad()`. The text encoder forward needs gradient tracking for its LoRA params. Current pattern:
```python
ctx = torch.no_grad() if not q_has_lora else contextlib.nullcontext()
with ctx:
    # encode_prompt calls
```
VAE encoding ALWAYS stays in `no_grad()` -- VAE is frozen and always has BatchNorm in eval mode during training.

### 3. No Model Wrapping — Direct `PeftModel` Access
Without accelerate's `prepare()`, models are never wrapped in DDP/DeepSpeed. `isinstance(model, PeftModel)` works directly. `save_pretrained()` works directly.

### 4. Optimizer Creation Must Be Conditional on `lora_target`
If `lora_target="qwen"`, the transformer has no LoRA -> no trainable params -> `create_optimizer` raises `ValueError`. Same for `lora_target="transformer"` and Qwen. Create optimizers conditionally:
```python
transformer_opt = create_optimizer(transformer, ...) if t_has_lora else None
qwen_opt = create_optimizer(text_encoder, ...) if q_has_lora else None
```
Similarly, `clip_grad_norm_` on Qwen must be guarded by `if qwen_opt is not None`.

### 5. `lora_target` Must Be Validated
Invalid values like `"none"` silently train 1500 steps with no optimizer, saving checkpoints containing only frozen base model weights. Must validate early:
```python
if lora_target not in ("transformer", "qwen", "both"):
    raise ValueError(...)
```

### 6. Config Wiring -- Every Key Must Have a Consumer
Past bugs: `target_size` accepted by dataset constructor but never passed from config; `guidance_scale`/`num_inference_steps` in config.yaml not read by validate(). Every config key must be read somewhere. If you add a new config key, verify end-to-end that it reaches its consumer.

Also: code defaults for config values can silently diverge from config.yaml. Example: `qwen_lr` config value is `1e-4` but `train.py:216` has `config.get("qwen_lr", 1e-3)` — if the key is accidentally deleted from config, the fallback is 10x higher. Keep code defaults in sync with config.yaml.

### 7. VAE `eval()` in Validation AND Training
`load_inference_models` in `validate.py` must call `vae.eval()` AND `transformer.eval()` — both have BatchNorm layers whose running stats drift in training mode even under `no_grad()`, corrupting decode quality across multi-image validation runs. `vae.eval()` is also set in `train.py:117` after model loading.

### 8. Timestep Shape Is (B,) Not (B, 1)
The Flux2 transformer expects 1D timestep tensors of shape (B,). Passing `.unsqueeze(1)` -- shape (B, 1) -- will silently cause shape errors downstream.

### 9. Device Placement — `transformer_device` vs `qwen_device`
VAE + transformer go to `transformer_device` (cuda:0). Qwen3 goes to `qwen_device` (cuda:1). `prompt_embeds` and `text_ids` must be `.to(transformer_device)` before passing to the transformer. Autograd handles gradient routing across devices via CopyBackwards. When both devices are the same (single-GPU mode), `.to()` is a no-op.

## Training Resume

Training can be resumed from a checkpoint via `python -m src.dehaze_lora.train --resume checkpoint-N`. The resume flow (`train.py` + `checkpoint.py`):

1. `load_training_state(checkpoint_dir)` loads `training_state.pt` — restores `global_step`, `micro_step`, optimizer states, and RNG state
2. `set_rng_state()` restores Python, NumPy, torch CPU, and CUDA RNG states (DataLoader worker RNG is replayed from seed, not captured)
3. LoRA weights reloaded via `PeftModel.from_pretrained()` onto the base models
4. If `global_step >= max_steps`, training exits immediately (already complete)
5. `checkpoint.py` also provides `save_checkpoint()` which writes `transformer_lora/`, `qwen_lora/`, `training_state.pt`, and `config.yaml`

## Test Structure

```
tests/
  __init__.py
  conftest.py              # Fixtures, VRAM guards, GPU markers, synthetic models, cleanup
  test_model_loading.py    # GPU: real model loading, LoRA injection, forward passes
  test_training_step.py    # Synthetic + GPU: flow matching logic, gradient accumulation, optimizer steps
  test_lora_injection.py   # CPU: LoRA injection, save/load roundtrip, merge/unload, target selection
  test_checkpoint.py       # GPU: checkpoint file structure, roundtrip weights, forward consistency
  test_cfg_inference.py    # CPU: CFG formula, token concat, scheduler integration, mock orchestration
  test_optimizer.py        # CPU: Muon optimizer creation, 2D validation, param changes
  test_dataset.py          # CPU: dataset loading, caption dropout, image transforms
  test_loss.py             # CPU: flow matching loss formula correctness
  test_vae.py              # GPU: VAE encode/decode roundtrip with real model
  test_tokenizer.py        # CPU: Qwen3 tokenizer behavior
  test_text_ids.py         # CPU: text ID generation shapes and values
  test_patchify.py         # CPU: patchify/unpatchify roundtrip, index values
  test_preprocess.py       # CPU: data preprocessing pipeline
  test_metrics.py          # CPU: PSNR/SSIM calculations
  test_utils.py            # CPU: config loading utilities
  test_validation.py       # CPU: validation split logic, subset selection, RNG isolation
  test_type_boundaries.py  # CPU: type coercion correctness at module boundaries
```

### Test Categories

- **CPU tests** (`-m "not gpu and not slow"`): Synthetic models, mock transformers, tiny operators. Run instantly, no GPU needed. These test all the logic gating, formulas, and edge cases.
- **GPU tests** (`-m gpu`): Load real Flux2 transformer or Qwen3 from HF hub. Each test loads one large model at a time, uses VRAM guards (`_require_vram_gb`), and cleans up with `del model; cleanup_gpu()`.
- **Slow tests** (`-m slow`): Tests that download or load real models. Skipped unless explicitly selected.

### Key Test Patterns

- **VRAM guard**: Tests that need real models call `_require_vram_gb(20)` or `_require_vram_gb(18)` to skip on insufficient GPU memory. Each model has known VRAM requirements: transformer ~18GB, Qwen3 ~16GB, VAE ~1GB.
- **Cleanup**: GPU tests MUST dereference models (`del model`) before calling `cleanup_gpu()` -- Python's GC cannot collect objects still referenced in local scope. The `module = None` pattern is required inside `finally` blocks.
- **Tiny inputs**: GPU forward-pass tests use 4x4 latents (16 tokens) to minimize activation memory.
- **Synthetic models**: `TinyAttention` and `TinyQwenAttention` in `conftest.py` provide minimal modules with correct LoRA target module names for injection/merge tests without loading real models.
- **Seed reset**: `conftest.py` has an `autouse=True` fixture that calls `torch.manual_seed(42)` and `torch.cuda.manual_seed_all(42)` before every test.

## Muon Optimizer Constraints

- All trainable parameters MUST be 2D tensors. LoRA A/B matrices naturally satisfy this.
- `create_optimizer` validates this and raises `ValueError` for non-2D params or no trainable params.
- Uses `adjust_lr_fn="match_rms_adamw"` (NOT the default `"original"`).
- Default momentum=0.95, weight_decay=0.01.
- If you add non-LoRA trainable params (e.g., a bias or norm layer), they MUST be 2D or you need to restructure the optimizer.

## VRAM per `lora_target` (batch_size=1, bf16, FlashAttn)

Actual model configs from HF hub snapshots:

| Model | Architecture | Params (bf16) |
|-------|-------------|---------------|
| Transformer | 8 double + 24 single blocks, hidden=4096 | ~18 GB |
| Qwen3 | 36 layers, hidden=4096, intermediate=12288 | ~17 GB |
| VAE | 32ch latent, 2×2 patchify, BN | ~0.2 GB |
| **Base total** | always loaded, always frozen | **~35 GB** |

LoRA (rank=16, alpha=8):
| Model | Target modules | Params |
|-------|---------------|--------|
| Transformer | to_q, to_k, to_v, to_out.0, to_qkv_mlp_proj (~184 modules) | ~24M (~48 MB) |
| Qwen3 | q_proj, k_proj, v_proj, o_proj (144 modules) | 19M (38 MB) |
| Optimizer + grads (fp32) | — | ~5× param count |

### Per-mode peak VRAM

Gradient checkpointing is enabled by default (`gradient_checkpointing: true` in config.yaml), reducing transformer activations from ~8 GB to ~2 GB:

| Component | `transformer` | `qwen` | `both` |
|-----------|:-----------:|:-----:|:-----:|
| Base weights | 35 GB | 35 GB | 35 GB |
| Transformer activations | ~2 GB | ~2 GB | ~2 GB |
| Qwen3 activations | 0 | ~2 GB | ~2 GB |
| LoRA + opt + grads | ~0.3 GB | ~0.2 GB | ~0.4 GB |
| CUDA overhead | ~3 GB | ~3 GB | ~3 GB |
| **Peak** | **~40 GB** | **~42 GB** | **~42 GB** |
| **2x40GB Peak (GPU0/GPU1)** | **~22 GB / ~20 GB** | **~22 GB / ~22 GB** | **~22 GB / ~22 GB** |

### Why transformer activations always cost ~2 GB (with gradient checkpointing)

All three modes need backprop through the transformer:
- `transformer` mode: transformer LoRA params need gradients → activations saved
- `qwen` mode: Qwen3 LoRA gradients must flow back through the transformer → activations saved (prompt_embeds from Qwen3 requires grad)
- `both` mode: both paths need gradients

The only VRAM saved in `transformer` mode is Qwen3 activations (~2 GB), because `encode_prompt` runs inside `torch.no_grad()` when Qwen3 has no LoRA (`train.py` encode_prompt block).

Without gradient checkpointing, transformer activations would be ~8 GB, adding ~6 GB across all modes. When gradient checkpointing is off, `transformer` mode is the safest choice for a single 80 GB GPU.

### Flash Attention requirement

Without Flash Attention, each transformer attention layer stores a 32-head × 2560² attention matrix (~420 MB per block × 32 blocks = ~13 GB). With Flash Attention this is eliminated — only Q/K/V projections are stored. The above estimates assume Flash Attention is active.

Activations scale roughly linearly with batch_size. With bs=4 (current default), `both` mode peaks at ~54 GB, leaving ~26 GB headroom on 80 GB. For single-GPU override (bs=2), `both` mode peaks at ~46 GB.

## Code Conventions

- **dtype**: `torch.bfloat16` throughout for all model weights and activations. VAE BN stats cast to match latent dtype/device.
- **Device management**: Transformer+VAE on `transformer_device` (cuda:0), Qwen3 on `qwen_device` (cuda:1). Cross-device autograd via `.to()` CopyBackwards — no manual gradient sync. Single-GPU: set both devices to `"cuda:0"` — `.to()` is a no-op.
- **Seed determinism**: Master seed (42) with `micro_step` added for per-batch unique RNG. `torch.backends.cudnn.deterministic = True` and `torch.backends.cudnn.benchmark = False` for deterministic GPU ops. DataLoader workers seed from `seed + worker_id`.
- **Imports**: No wildcard imports. Module-level imports in standard order: stdlib -> third-party -> project.
- **Type annotations**: Partial — public function signatures are annotated, internal helpers may not be. Not enforced by a type checker.
- **Config format**: YAML at `configs/config.yaml`, loaded via `utils.load_config`. All values accessible via `config.get("key", default)` pattern.
- **Error handling**: Explicit `ValueError` for invalid config values, empty datasets, and missing files. No silent defaults that could hide misconfiguration.

## Adding New Features

1. **New config key?** Add to `configs/config.yaml`, add `config.get("key", default)` in the consumer, and verify end-to-end wiring.
2. **New LoRA target?** Add module names to `QWEN_LORA_MODULES` or `TRANSFORMER_LORA_MODULES` in `model.py`. Verify they exist on the real model.
3. **Changing training logic?** Add a synthetic CPU test first that exercises the new path. The synthetic tests catch silent failures without needing GPU time.
4. **New model component?** Respect the VRAM limits -- load one large model per test, never keep transformer + Qwen3 in GPU simultaneously in tests.
5. **Use raw PyTorch APIs**: Use `loss.backward()`, `torch.nn.utils.clip_grad_norm_()`, and `torch.amp.autocast("cuda", dtype=torch.bfloat16)`. Forward passes wrap with `sdpa_kernel(SDPBackend.FLASH_ATTENTION)`. Models are placed manually — no `prepare()` or `unwrap_model()`.
