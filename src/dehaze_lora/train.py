import contextlib
import random

import numpy as np
import torch
from pathlib import Path
from torch.nn.attention import SDPBackend, sdpa_kernel
from torch.utils.data import DataLoader
from tqdm import tqdm

from .model import (
    load_models, encode_prompts, encode_vae_image,
    patchify_and_make_ids, unpatchify,
)
from .loss import flow_matching_loss
from .optimizer import create_optimizer
from peft import PeftModel



def save_checkpoint(transformer, text_encoder, step, output_dir):
    ckpt_dir = Path(output_dir) / f"checkpoint-{step}"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    transformer.save_pretrained(ckpt_dir / "transformer_lora")

    if isinstance(text_encoder, PeftModel):
        text_encoder.save_pretrained(ckpt_dir / "qwen_lora")

    print(f"Checkpoint saved: {ckpt_dir}")


def train(config: dict, output_dir: str = "outputs/checkpoints"):
    transformer_device = config.get("transformer_device", "cuda:0")
    qwen_device = config.get("qwen_device", "cuda:1")

    # ---- Fixed seed ----
    seed = config.get("seed", 42)
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
        model_name=config["model_name"],
        lora_target=config.get("lora_target", "both"),
        lora_rank=config.get("lora_rank", 16),
        lora_alpha=config.get("lora_alpha", 8),
        transformer_device=transformer_device,
        qwen_device=qwen_device,
        gradient_checkpointing=config.get("gradient_checkpointing", False),
    )
    vae = models["vae"]
    transformer = models["transformer"]
    scheduler = models["scheduler"]
    text_encoder = models["text_encoder"]
    tokenizer = models["tokenizer"]
    vae.eval()

    # Read transformer config
    transformer_config = transformer.config
    patch_size = getattr(transformer_config, "patch_size", 1)

    # ---- Dataset ----
    from .dataset import DehazeDataset

    train_dataset = DehazeDataset(
        metadata_path=config["train_metadata"],
        caption_dropout_rate=config.get("caption_dropout_rate", 0.1),
    )
    if len(train_dataset) == 0:
        raise ValueError(
            f"Training dataset is empty: {config['train_metadata']}"
        )

    def _seed_worker(worker_id):
        worker_seed = seed + worker_id
        random.seed(worker_seed)
        np.random.seed(worker_seed)
        torch.manual_seed(worker_seed)

    dl_generator = torch.Generator().manual_seed(seed)
    train_loader = DataLoader(
        train_dataset,
        batch_size=config["batch_size"],
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        generator=dl_generator,
        worker_init_fn=_seed_worker,
    )

    # ---- Optimizers ----
    lora_target = config.get("lora_target", "both")
    if lora_target not in ("transformer", "qwen", "both"):
        raise ValueError(
            f"Invalid lora_target={lora_target!r}, "
            f"expected one of: transformer, qwen, both"
        )
    transformer_lr = float(config.get("transformer_lr", 1e-3))
    qwen_lr = float(config.get("qwen_lr", 1e-3))
    wd = float(config.get("weight_decay", 0.01))
    momentum = float(config.get("momentum", 0.95))
    transformer_grad_clip = float(config.get("transformer_grad_clip", 1.0))
    qwen_grad_clip = float(config.get("qwen_grad_clip", 1.0))

    t_has_lora = lora_target in ("transformer", "both")
    q_has_lora = lora_target in ("qwen", "both")

    transformer_opt = create_optimizer(
        transformer, lr=transformer_lr, weight_decay=wd, momentum=momentum
    ) if t_has_lora else None
    qwen_opt = create_optimizer(
        text_encoder, lr=qwen_lr, weight_decay=wd, momentum=momentum
    ) if q_has_lora else None

    max_seq_len = config.get("max_sequence_length", 512)
    global_step = 0
    micro_step = 0
    max_steps = config["max_steps"]
    save_every = config.get("save_every", 250)
    log_freq = config.get("log_freq", 10)
    if save_every <= 0 or log_freq <= 0:
        raise ValueError(
            f"save_every ({save_every}) and log_freq ({log_freq}) must be > 0"
        )
    grad_accum = config["gradient_accumulation_steps"]

    # ---- wandb ----
    import wandb

    exp_name = (
        f"{lora_target}_"
        f"r{config.get('lora_rank', 16)}_"
        f"tlr{transformer_lr}_qlr{qwen_lr}_"
        f"seed{seed}"
    )
    wandb.init(
        project=config.get("wandb_project", "dehaze-flux2-klein"),
        entity=config.get("wandb_entity"),
        mode=config.get("wandb_mode", "online"),
        name=exp_name,
        config=config,
    )

    print(
        f"Training: {max_steps} steps, batch={config['batch_size']}, "
        f"accum={grad_accum}, effective_batch={config['batch_size'] * grad_accum}"
    )
    print(f"Devices: transformer={transformer_device}, qwen={qwen_device}")

    step_uncond = 0
    step_total = 0

    with tqdm(total=max_steps, desc="Training", unit="step", dynamic_ncols=True) as pbar:
        while global_step < max_steps:
            for batch in train_loader:
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

                # Timestep and noise
                bsz = gt_latent.shape[0]
                rng = torch.Generator(transformer_device).manual_seed(seed + micro_step)
                t = torch.randint(
                    0, scheduler.config.num_train_timesteps, (bsz,),
                    generator=rng, device=transformer_device,
                ).long()
                sigmas = t.float() / scheduler.config.num_train_timesteps

                noise = torch.randn(
                    gt_latent.shape, generator=rng,
                    device=transformer_device, dtype=torch.bfloat16,
                )
                noisy_latent = (
                    (1.0 - sigmas.view(bsz, 1, 1, 1)) * gt_latent
                    + sigmas.view(bsz, 1, 1, 1) * noise
                )

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

                    loss = flow_matching_loss(v_theta, gt_latent, noise)
                    loss = loss / grad_accum  # normalize for accumulation

                loss.backward()
                micro_step += 1

                if micro_step % grad_accum == 0:
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
                    pbar.update(1)
                    pbar.set_postfix(loss=f"{loss.item():.4f}")

                    if global_step % log_freq == 0:
                        cond_ratio = 1.0 - step_uncond / max(step_total, 1)
                        log_dict = {
                            "train/loss": loss.item(),
                            "train/cond_ratio": cond_ratio,
                            "train/transformer_lr": transformer_lr,
                            "train/step": global_step,
                        }
                        step_uncond = 0
                        step_total = 0
                        if qwen_opt is not None:
                            log_dict["train/qwen_lr"] = qwen_lr
                        wandb.log(log_dict, step=global_step)

                if global_step > 0 and global_step % save_every == 0:
                    save_checkpoint(
                        transformer, text_encoder,
                        global_step, output_dir,
                    )

                if global_step >= max_steps:
                    break

    wandb.finish()
    print("Training complete.")


if __name__ == "__main__":
    import argparse
    from .utils import load_config

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()

    config = load_config(args.config)
    train(config, output_dir=config.get("output_dir", "outputs/checkpoints"))
