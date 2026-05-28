# Validation Integration During Training

## Context

Training has no validation loop. Validation (`validate.py`) is a standalone script that must be run manually on saved checkpoints. The user wants periodic validation during training with three inference modes per sample: conditional generation, reconstruction (unconditional), and CFG.

## Requirements

### Data Split
- Deterministic 95/5 train/val split from RESIDE metadata at training start
- Split based on `hash(image_path) % 100 < 5`, seeded for reproducibility
- Training DataLoader must never see validation samples

### Validation Trigger
- Run at every `save_every` step, BEFORE saving the checkpoint
- Also run at the final step if it wasn't already a save_every multiple

### Validation Subset
- 4 images per validation run, deterministically selected
- Subset seed = `seed + global_step` for step-level reproducibility

### Three Inference Modes (shared computation)
Per validation image, VAE encode hazy once, text encode cond+uncond once. Each denoising timestep runs cond + uncond forward passes (2 total), yielding 3 outputs:

| Mode | Velocity | Purpose |
|------|----------|---------|
| Conditional | v_cond | Dehaze mapping with prompt |
| Reconstruction | v_uncond | Pre-training behavior preservation (empty prompt) |
| CFG | v_uncond + gs*(v_cond - v_uncond) | Deployment mode, compute PSNR/SSIM vs GT |

Total: 4 images x 3 modes = 12 output images per validation run.

### wandb Logging
- 12 output images as comparison grids (hazy / GT / cond / recon / CFG)
- PSNR/SSIM curves from CFG mode
- Images uploaded as wandb Images, not saved to disk

### NH-HAZE Evaluation
- Keep existing `validate.py` standalone entry point
- Add `scripts/test.sh` for OOD evaluation on NH-HAZE

## Architecture

### File Changes

#### `src/dehaze_lora/validate.py` — Refactor
- Extract `_denoise_cfg()`: shared denoising loop that runs cond+uncond forward per timestep, returns 3 velocity fields. Replace current `dehaze_single()` body with this.
- Keep `dehaze_single()` as thin wrapper (still used for standalone NH-HAZE eval).
- Add `run_validation_batch()`: iterates over val subset, collects 12 outputs + PSNR/SSIM, returns results dict.
- `validate()` function unchanged (standalone entry point).

#### `src/dehaze_lora/train.py` — Add validation hook
- After model loading: split metadata into train/val, create `DehazeDataset` with filtered train items only
- `DehazeDataset` accepts optional `metadata_items: list | None` to override file reading
- Before `save_checkpoint`: call `run_validation_batch()`, log to wandb
- Validation uses training models directly (no re-loading) with `eval()` + `no_grad()`
- After validation: restore models to `train()` mode

#### `src/dehaze_lora/dataset.py` — Minor
- `DehazeDataset.__init__` accepts optional `metadata_items: list | None`

#### `configs/config.yaml` — New keys
```yaml
val_split: 0.05
val_subset_size: 4
val_guidance_scale: 3.5
val_num_inference_steps: 28
```

#### `scripts/validate_nhhaze.sh` — New
- Shell wrapper for standalone NH-HAZE evaluation

### Validation Flow (per save_every step)

```
1. Set vae.eval(), transformer.eval(), text_encoder.eval()
2. Select 4 val images deterministically (seed + global_step)
3. For each image:
   a. VAE encode hazy (once)
   b. Text encode cond + uncond (once each)
   c. Generate shared noise (fixed seed)
   d. Denoising loop (num_inference_steps timesteps):
      - cond forward → v_cond
      - uncond forward → v_uncond
      - Mode 1 (cond): use v_cond
      - Mode 2 (recon): use v_uncond
      - Mode 3 (CFG): use v_uncond + gs*(v_cond - v_uncond)
   e. VAE decode 3 outputs
   f. Compute PSNR/SSIM for CFG output vs GT
4. Log results to wandb
5. Restore model train() modes
6. Save checkpoint
```

### VRAM Safety
- Entire validation wrapped in `torch.no_grad()` — no activations saved, no gradient buffers allocated
- Text encodings computed once per image (not per timestep), reused across all denoising steps
- VAE encode computed once per image, reused across all 3 modes
- Models set to `eval()` before validation, restored to `train()` after

### Determinism Guarantees
- Data split: `hash(path) % 100 < 5` — independent of execution order
- Subset selection: `seed + global_step` — pure function of step counter
- Noise: fixed seed per validation run — same noise for all 3 modes (comparable)
- RNG isolation: capture/restore training RNG state around validation

## Edge Cases
- `DehazeDataset` with `metadata_items=None` (full file read): backward-compatible
- Validation at step 0 (before any training): skip if `global_step == 0`
- Empty val split (very small dataset): warn, skip validation
- OOM during validation: validation images are small (4 x 28 steps), should fit comfortably
