from __future__ import annotations

import contextlib
import hashlib
import json
import random
from pathlib import Path
from typing import Any, Iterator, Mapping, Optional

import numpy as np
import torch
from torch.nn.attention import SDPBackend, sdpa_kernel
from torch.utils.data import DataLoader
from tqdm import tqdm

from .model import (
    load_models, encode_prompts, encode_vae_image,
    patchify_and_make_ids, unpatchify,
)
from .loss import flow_matching_loss
from .optimizer import create_optimizer
from .checkpoint import (
    get_rng_state, set_rng_state, save_checkpoint, load_training_state,
)
from .validate import run_validation_batch
from .types import MetadataItem
from peft import PeftModel


def _split_train_val_metadata(
    metadata: list[MetadataItem],
    val_split: float,
) -> tuple[list[MetadataItem], list[MetadataItem]]:
    threshold = int(float(val_split) * 100)
    train_items: list[MetadataItem] = []
    val_items: list[MetadataItem] = []
    for item in metadata:
        bucket = int(hashlib.md5(item["gt"].encode()).hexdigest(), 16) % 100
        if bucket < threshold:
            val_items.append(item)
        else:
            train_items.append(item)
    return train_items, val_items


def _sample_sigmas_and_noise(
    shape: torch.Size,
    batch_size: int,
    shift: float,
    seed: int,
    micro_step: int,
    device: str,
    dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor]:
    rng = torch.Generator(device).manual_seed(int(seed) + int(micro_step))
    z = torch.randn(batch_size, generator=rng, device=device, dtype=torch.float32)
    sigma_raw = torch.sigmoid(z)
    sigmas = (sigma_raw * float(shift)) / (1 + (float(shift) - 1) * sigma_raw)
    noise = torch.randn(shape, generator=rng, device=device, dtype=dtype)
    return sigmas, noise


def _make_noisy_latent(
    target_latent: torch.Tensor,
    noise: torch.Tensor,
    sigmas: torch.Tensor,
) -> torch.Tensor:
    bsz = target_latent.shape[0]
    return (
        (1.0 - sigmas.view(bsz, 1, 1, 1)) * target_latent
        + sigmas.view(bsz, 1, 1, 1) * noise
    )


def _training_batches(
    train_loader: DataLoader,
    skip: int = 0,
) -> Iterator[dict[str, torch.Tensor]]:
    """Yield batches from DataLoader, skipping `skip` batches first.

    Handles epoch boundaries via StopIteration, creating a new iterator
    (and therefore a new shuffle permutation) each epoch. After skip,
    the generator state and per-worker RNG state are identical to what
    they would have been after skip batches in a fresh training run.
    """
    it = iter(train_loader)
    for _ in range(skip):
        try:
            next(it)
        except StopIteration:
            it = iter(train_loader)
            next(it)
    while True:
        try:
            yield next(it)
        except StopIteration:
            it = iter(train_loader)
            yield next(it)


def _select_val_subset(
    val_metadata: list[MetadataItem],
    k: int,
    seed: int,
) -> list[MetadataItem]:
    """Deterministic subset selection from validation set."""
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


def _lora_target_flags(lora_target: str) -> tuple[bool, bool]:
    if lora_target not in ("transformer", "qwen", "both"):
        raise ValueError(
            f"Invalid lora_target={lora_target!r}, "
            f"expected one of: transformer, qwen, both"
        )
    return lora_target in ("transformer", "both"), lora_target in ("qwen", "both")


def _make_wandb_run_name(
    lora_target: str,
    lora_rank: int,
    transformer_lr: float,
    qwen_lr: float,
    seed: int,
) -> str:
    t_has_lora, q_has_lora = _lora_target_flags(lora_target)
    parts = [lora_target, f"r{int(lora_rank)}"]
    if t_has_lora:
        parts.append(f"tlr{transformer_lr}")
    if q_has_lora:
        parts.append(f"qlr{qwen_lr}")
    parts.append(f"seed{seed}")
    return "_".join(parts)


def _build_train_log_dict(
    avg_loss: float,
    cond_ratio: float,
    global_step: int,
    transformer_opt: Any,
    qwen_opt: Any,
) -> dict[str, float | int]:
    log_dict: dict[str, float | int] = {
        "train/loss": avg_loss,
        "train/cond_ratio": cond_ratio,
        "train/step": global_step,
    }
    if transformer_opt is not None:
        log_dict["train/transformer_lr"] = transformer_opt.param_groups[0]["lr"]
    if qwen_opt is not None:
        log_dict["train/qwen_lr"] = qwen_opt.param_groups[0]["lr"]
    return log_dict


def _set_training_modes(
    transformer: Any,
    text_encoder: Any,
    t_has_lora: bool,
    q_has_lora: bool,
) -> None:
    if t_has_lora:
        transformer.train()
    else:
        transformer.eval()
    if q_has_lora:
        text_encoder.train()
    else:
        text_encoder.eval()


def train(
    config: dict[str, Any],
    output_dir: str = "outputs/checkpoints",
    resume_from: Optional[str] = None,
) -> None:
    transformer_device = str(config.get("transformer_device", "cuda:0"))
    qwen_device = str(config.get("qwen_device", "cuda:1"))
    lora_target = str(config.get("lora_target", "both"))
    t_has_lora, q_has_lora = _lora_target_flags(lora_target)

    # ---- Fixed seed ----
    seed = int(config.get("seed", 42))
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # ---- Flash Attention check ----
    if torch.backends.cuda.flash_sdp_enabled():
        print("Flash SDPA backend: enabled (PyTorch built-in)")
    else:
        print(
            "\033[1;33mWARNING: Flash SDPA not available!\033[0m "
            "Falling back to memory-efficient attention."
        )

    # ---- Load models ----
    models = load_models(
        model_name=str(config["model_name"]),
        lora_target=lora_target,
        lora_rank=int(config.get("lora_rank", 16)),
        lora_alpha=int(config.get("lora_alpha", 8)),
        transformer_device=transformer_device,
        qwen_device=qwen_device,
        gradient_checkpointing=bool(config.get("gradient_checkpointing", False)),
    )
    vae = models["vae"]
    transformer = models["transformer"]
    scheduler = models["scheduler"]
    text_encoder = models["text_encoder"]
    tokenizer = models["tokenizer"]
    vae.eval()
    _set_training_modes(transformer, text_encoder, t_has_lora, q_has_lora)

    # Read transformer config
    transformer_config = transformer.config
    patch_size = int(getattr(transformer_config, "patch_size", 1))

    # ---- Resume ----
    if resume_from is not None:
        ckpt_dir = Path(resume_from)
        state = load_training_state(ckpt_dir)
        # Restore RNG states (must happen after seed setup, before DataLoader)
        set_rng_state(state["rng_states"])
        global_step = state["global_step"]
        micro_step = state["micro_step"]
        print(f"Resuming from {ckpt_dir} at global_step={global_step}, micro_step={micro_step}")

        # Reload LoRA weights
        base_transformer = transformer.get_base_model() if isinstance(transformer, PeftModel) else transformer
        base_qwen = text_encoder.get_base_model() if isinstance(text_encoder, PeftModel) else text_encoder
        if t_has_lora and (ckpt_dir / "transformer_lora").exists():
            transformer = PeftModel.from_pretrained(base_transformer, str(ckpt_dir / "transformer_lora"))
            transformer.to(transformer_device)
        if q_has_lora and (ckpt_dir / "qwen_lora").exists():
            text_encoder = PeftModel.from_pretrained(base_qwen, str(ckpt_dir / "qwen_lora"))
            text_encoder.to(qwen_device)
        _set_training_modes(transformer, text_encoder, t_has_lora, q_has_lora)

        if global_step >= config["max_steps"]:
            print(f"Resumed at step {global_step} >= max_steps={config['max_steps']}. Training already complete.")
            return
    else:
        global_step = 0
        micro_step = 0

    # ---- Dataset with train/val split ----
    from .dataset import DehazeDataset

    all_metadata = [json.loads(l) for l in open(config["train_metadata"])]
    val_split = float(config.get("val_split", 0.05))
    train_items, val_items = _split_train_val_metadata(all_metadata, val_split)

    if len(train_items) == 0:
        raise ValueError(
            f"No training samples after {val_split=} split "
            f"({len(all_metadata)} total)"
        )
    if len(val_items) == 0:
        print(
            f"\033[1;33mWARNING: Validation set is empty "
            f"({val_split=}, {len(all_metadata)} total). "
            "Skipping validation.\033[0m"
        )

    print(
        f"Data split: {len(train_items)} train, "
        f"{len(val_items)} val ({val_split:.0%})"
    )

    train_dataset = DehazeDataset(
        metadata_path=config["train_metadata"],
        caption_dropout_rate=float(config.get("caption_dropout_rate", 0.1)),
        dropout_seed=seed,
        metadata_items=train_items,
    )
    if len(train_dataset) == 0:
        raise ValueError(
            f"Training dataset is empty: {config['train_metadata']}"
        )

    def _seed_worker(worker_id: int) -> None:
        worker_seed = int(seed + worker_id)
        random.seed(worker_seed)
        np.random.seed(worker_seed)
        torch.manual_seed(worker_seed)

    dl_generator = torch.Generator().manual_seed(seed)
    train_loader = DataLoader(
        train_dataset,
        batch_size=int(config["batch_size"]),
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        generator=dl_generator,
        worker_init_fn=_seed_worker,
    )

    # ---- Optimizers ----
    transformer_lr = float(config.get("transformer_lr", 1e-3))
    qwen_lr = float(config.get("qwen_lr", 1e-4))
    wd = float(config.get("weight_decay", 0.01))
    momentum = float(config.get("momentum", 0.95))
    transformer_grad_clip = float(config.get("transformer_grad_clip", 1.0))
    qwen_grad_clip = float(config.get("qwen_grad_clip", 1.0))
    warmup_steps = int(config.get("warmup_steps", 0))

    transformer_opt = create_optimizer(
        transformer, lr=transformer_lr, weight_decay=wd, momentum=momentum
    ) if t_has_lora else None
    qwen_opt = create_optimizer(
        text_encoder, lr=qwen_lr, weight_decay=wd, momentum=momentum
    ) if q_has_lora else None

    # Load optimizer states if resuming
    if resume_from is not None:
        opt_states = state["optimizer_states"]
        if opt_states["transformer"] is not None and transformer_opt is not None:
            transformer_opt.load_state_dict(opt_states["transformer"])
        if opt_states["qwen"] is not None and qwen_opt is not None:
            qwen_opt.load_state_dict(opt_states["qwen"])

    max_seq_len = int(config.get("max_sequence_length", 512))
    max_steps = int(config["max_steps"])
    save_every = int(config.get("save_every", 250))
    log_freq = int(config.get("log_freq", 10))
    if save_every <= 0 or log_freq <= 0:
        raise ValueError(
            f"save_every ({save_every}) and log_freq ({log_freq}) must be > 0"
        )
    grad_accum = int(config["gradient_accumulation_steps"])

    # ---- wandb ----
    import wandb

    exp_name = _make_wandb_run_name(
        lora_target=lora_target,
        lora_rank=int(config.get("lora_rank", 16)),
        transformer_lr=transformer_lr,
        qwen_lr=qwen_lr,
        seed=seed,
    )
    wandb.init(
        project=str(config.get("wandb_project", "dehaze-flux2-klein")),
        entity=config.get("wandb_entity"),
        mode=str(config.get("wandb_mode", "online")),
        name=exp_name,
        config=config,
    )

    def _validate_and_save(save: bool = True) -> None:
        nonlocal global_step, micro_step

        # Run validation (before saving checkpoint)
        if len(val_items) > 0:
            rng_state_before = get_rng_state()

            val_subset = _select_val_subset(
                val_items,
                k=int(config.get("val_subset_size", 4)),
                seed=int(seed + global_step),
            )

            # Models to eval mode for validation
            transformer.eval()
            text_encoder.eval()

            val_results = run_validation_batch(
                vae=vae,
                transformer=transformer,
                scheduler=scheduler,
                text_encoder=text_encoder,
                tokenizer=tokenizer,
                val_subset=val_subset,
                guidance_scale=float(config.get("val_guidance_scale", config.get("guidance_scale", 3.5))),
                num_inference_steps=int(config.get("val_num_inference_steps", config.get("num_inference_steps", 28))),
                max_seq_len=max_seq_len,
                transformer_device=transformer_device,
                qwen_device=qwen_device,
                seed=int(seed + global_step),
            )

            # Restore train mode
            _set_training_modes(transformer, text_encoder, t_has_lora, q_has_lora)
            # VAE stays eval (never trained, BatchNorm in eval)

            # Restore RNG (validation consumed RNG state)
            set_rng_state(rng_state_before)

            # Log validation results to wandb
            if val_results["images"]:
                wandb_images = []
                for j, img_dict in enumerate(val_results["images"]):
                    comparison = torch.cat([
                        img_dict["hazy"],
                        img_dict["gt"],
                        img_dict["cond"],
                        img_dict["uncond"],
                        img_dict["cfg"],
                    ], dim=2)  # horizontal concatenation
                    comparison = (
                        comparison.detach()
                        .float()
                        .clamp(0, 1)
                        .mul(255)
                        .byte()
                        .permute(1, 2, 0)
                        .cpu()
                        .numpy()
                    )
                    wandb_images.append(
                        wandb.Image(comparison, caption=f"val_{j}")
                    )
                wandb.log({
                    "val/images": wandb_images,
                    "val/psnr": sum(val_results["psnr"]) / len(val_results["psnr"]),
                    "val/ssim": sum(val_results["ssim"]) / len(val_results["ssim"]),
                    "val/step": global_step,
                }, step=global_step)
                print(
                    f"Validation (step {global_step}) - "
                    f"PSNR: {sum(val_results['psnr'])/len(val_results['psnr']):.2f} dB | "
                    f"SSIM: {sum(val_results['ssim'])/len(val_results['ssim']):.4f}"
                )

        # Save checkpoint
        if save:
            rng_state = get_rng_state()
            save_checkpoint(
                transformer, text_encoder, global_step, output_dir,
                global_step=global_step, micro_step=micro_step,
                rng_state=rng_state,
                transformer_opt=transformer_opt, qwen_opt=qwen_opt,
                config=config,
            )

    print(
        f"Training: {max_steps} steps, batch={int(config['batch_size'])}, "
        f"accum={grad_accum}, effective_batch={int(config['batch_size']) * grad_accum}"
    )
    print(f"Devices: transformer={transformer_device}, qwen={qwen_device}")

    step_uncond = 0
    step_total = 0
    step_loss = 0.0

    # Sanity validation before any training (step 0)
    _validate_and_save(save=False)

    with tqdm(
        total=max_steps, initial=global_step,
        desc="Training", unit="step", dynamic_ncols=True,
    ) as pbar:
        for batch in _training_batches(train_loader, skip=micro_step):
            with torch.no_grad():
                hazy = batch["hazy"].to(transformer_device, dtype=torch.bfloat16)
                gt = batch["gt"].to(transformer_device, dtype=torch.bfloat16)

                hazy_latent = encode_vae_image(vae, hazy)
                gt_latent = encode_vae_image(vae, gt)

            # Encode text (caption dropout already handled by dataset)
            captions = batch["caption"]
            is_uncond = [c == "" for c in captions]
            step_uncond += sum(is_uncond)
            step_total += len(is_uncond)

            ctx = torch.no_grad() if not q_has_lora else contextlib.nullcontext()
            with ctx:
                prompt_embeds, text_ids = encode_prompts(
                    text_encoder, tokenizer, captions,
                    max_seq_len, qwen_device, torch.bfloat16,
                )
            prompt_embeds = prompt_embeds.to(transformer_device)
            text_ids = text_ids.to(transformer_device)

            # Timestep and noise (logit-normal + shift, matching Flux2 Klein training)
            # RNG: Generator(seed + micro_step) — deterministic per micro-batch
            shift = scheduler.config.shift  # 3.0 for Flux2 Klein Base
            sigmas, noise = _sample_sigmas_and_noise(
                gt_latent.shape, gt_latent.shape[0], shift,
                seed, micro_step, transformer_device, torch.bfloat16,
            )
            # Caption dropout removes only text. The flow target remains GT clear image Y,
            # so empty prompt learns p(Y | I), not p(I | I).
            target_latent = gt_latent
            noisy_latent = _make_noisy_latent(target_latent, noise, sigmas)

            # Patchify
            noisy_tokens, noisy_ids = patchify_and_make_ids(
                noisy_latent, patch_size=patch_size, index=0.0,
            )
            ref_tokens, ref_ids = patchify_and_make_ids(
                hazy_latent, patch_size=patch_size, index=10.0,
            )
            hidden_states = torch.cat([noisy_tokens, ref_tokens], dim=1)
            img_ids = torch.cat([noisy_ids, ref_ids], dim=1)

            # Forward + backward with manual gradient accumulation (force Flash SDPA)
            with (
                sdpa_kernel(SDPBackend.FLASH_ATTENTION),
                torch.amp.autocast("cuda", dtype=torch.bfloat16),
            ):
                model_pred = transformer(
                    hidden_states=hidden_states,
                    encoder_hidden_states=prompt_embeds,
                    timestep=sigmas.to(dtype=torch.bfloat16),
                    img_ids=img_ids,
                    txt_ids=text_ids,
                    return_dict=False,
                )[0]

                v_theta = model_pred[:, : noisy_tokens.shape[1], :]
                v_theta = unpatchify(
                    v_theta,
                    gt_latent.shape[2], gt_latent.shape[3],
                    patch_size=patch_size,
                )

                loss = flow_matching_loss(v_theta, target_latent, noise)
                loss_scaled = loss / grad_accum  # normalize for accumulation

            loss_scaled.backward()
            step_loss += loss.detach()
            micro_step += 1

            if micro_step % grad_accum == 0:
                if warmup_steps > 0:
                    warmup_factor = min(1.0, (global_step + 1) / warmup_steps)
                    if transformer_opt is not None:
                        transformer_opt.param_groups[0]['lr'] = transformer_lr * warmup_factor
                    if qwen_opt is not None:
                        qwen_opt.param_groups[0]['lr'] = qwen_lr * warmup_factor

                if transformer_opt is not None:
                    torch.nn.utils.clip_grad_norm_(
                        transformer.parameters(), transformer_grad_clip
                    )
                    transformer_opt.step()
                    transformer_opt.zero_grad()

                if qwen_opt is not None:
                    torch.nn.utils.clip_grad_norm_(
                        text_encoder.parameters(), qwen_grad_clip
                    )
                    qwen_opt.step()
                    qwen_opt.zero_grad()

                global_step += 1
                avg_loss = float((step_loss / grad_accum).item())
                pbar.update(1)
                pbar.set_postfix(loss=f"{avg_loss:.4f}")

                if global_step % log_freq == 0:
                    cond_ratio = float(1.0 - step_uncond / max(step_total, 1))
                    log_dict = _build_train_log_dict(
                        avg_loss=avg_loss,
                        cond_ratio=cond_ratio,
                        global_step=global_step,
                        transformer_opt=transformer_opt,
                        qwen_opt=qwen_opt,
                    )
                    wandb.log(log_dict, step=global_step)

                step_uncond = 0
                step_total = 0
                step_loss = 0.0

                if global_step > 0 and global_step % save_every == 0:
                    _validate_and_save()

                if global_step >= max_steps:
                    break

    if global_step % save_every != 0:
        _validate_and_save()

    wandb.finish()
    print("Training complete.")


if __name__ == "__main__":
    import argparse
    from .utils import load_config

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--resume", type=str, default=None,
                        help="Resume training from a checkpoint directory")
    args = parser.parse_args()

    config = load_config(args.config)
    train(
        config,
        output_dir=str(config.get("output_dir", "outputs/checkpoints")),
        resume_from=args.resume,
    )
