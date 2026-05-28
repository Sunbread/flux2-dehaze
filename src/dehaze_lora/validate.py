import torch
from torch.nn.attention import SDPBackend, sdpa_kernel
from torch.utils.data import DataLoader
from diffusers import AutoencoderKLFlux2, FlowMatchEulerDiscreteScheduler, Flux2Transformer2DModel
from peft import PeftModel
from tqdm import tqdm
import numpy as np
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
import json
import os
from pathlib import Path

from .model import (
    encode_prompt, encode_vae_image, decode_vae_image,
    patchify_and_make_ids, unpatchify,
)


def load_inference_models(
    model_name="black-forest-labs/FLUX.2-klein-base-9B",
    transformer_lora_path=None,
    qwen_lora_path=None,
    device="cuda",
):
    vae = AutoencoderKLFlux2.from_pretrained(
        model_name, subfolder="vae", torch_dtype=torch.bfloat16
    )
    vae.requires_grad_(False)
    vae.eval()
    vae.to(device)

    transformer = Flux2Transformer2DModel.from_pretrained(
        model_name, subfolder="transformer", torch_dtype=torch.bfloat16
    )
    if transformer_lora_path:
        transformer = PeftModel.from_pretrained(transformer, transformer_lora_path)
        print(f"Loaded transformer LoRA from {transformer_lora_path}")
    transformer.requires_grad_(False)
    transformer.to(device)
    transformer.eval()

    scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(
        model_name, subfolder="scheduler"
    )

    return vae, transformer, scheduler


@torch.no_grad()
def dehaze_single(
    vae,
    transformer,
    scheduler,
    text_encoder,
    tokenizer,
    hazy_image: torch.Tensor,
    prompt: str,
    guidance_scale: float = 3.5,
    num_inference_steps: int = 28,
    device: str = "cuda",
):
    """
    Dehaze inference with two-pass CFG.

    v_cfg = v_uncond + guidance_scale * (v_cond - v_uncond)

    Reference image (hazy) in both conditional and unconditional branches.
    Pure noise start, not from hazy latent.
    """
    bsz = hazy_image.shape[0]

    transformer_config = transformer.config
    patch_size = getattr(transformer_config, "patch_size", 1)

    # VAE encode reference image (patchify + BN normalize)
    hazy_latent = encode_vae_image(vae, hazy_image.to(device, dtype=torch.bfloat16))

    # Encode text for both conditional and unconditional branches
    cond_embeds, cond_text_ids = encode_prompt(
        text_encoder, tokenizer, prompt, 512, device, torch.bfloat16,
    )
    uncond_embeds, uncond_text_ids = encode_prompt(
        text_encoder, tokenizer, "", 512, device, torch.bfloat16,
    )

    # Reference image tokens (precompute once, same for both branches)
    ref_tokens, ref_ids = patchify_and_make_ids(
        hazy_latent, patch_size=patch_size, index=10.0,
    )

    # Start from pure noise
    scheduler.set_timesteps(num_inference_steps)
    z = torch.randn_like(hazy_latent)

    for t in tqdm(scheduler.timesteps, desc="Denoising"):
        noisy_tokens, noisy_ids = patchify_and_make_ids(
            z, patch_size=patch_size, index=0.0,
        )

        # Shared image tokens (noisy + ref)
        img_hidden = torch.cat([noisy_tokens, ref_tokens], dim=1)
        img_ids_combined = torch.cat([noisy_ids, ref_ids], dim=1)

        timestep = (
            t.float().unsqueeze(0).to(device=device, dtype=torch.bfloat16)
            / scheduler.config.num_train_timesteps
        )

        # Conditional forward
        with sdpa_kernel(SDPBackend.FLASH_ATTENTION):
            v_cond = transformer(
                hidden_states=img_hidden,
                encoder_hidden_states=cond_embeds,
                timestep=timestep,
                img_ids=img_ids_combined,
                txt_ids=cond_text_ids,
                return_dict=False,
            )[0]
        v_cond = v_cond[:, : noisy_tokens.shape[1], :]
        v_cond = unpatchify(
            v_cond, z.shape[2], z.shape[3], patch_size=patch_size,
        )

        # Unconditional forward
        with sdpa_kernel(SDPBackend.FLASH_ATTENTION):
            v_uncond = transformer(
                hidden_states=img_hidden,
                encoder_hidden_states=uncond_embeds,
                timestep=timestep,
                img_ids=img_ids_combined,
                txt_ids=uncond_text_ids,
                return_dict=False,
            )[0]
        v_uncond = v_uncond[:, : noisy_tokens.shape[1], :]
        v_uncond = unpatchify(
            v_uncond, z.shape[2], z.shape[3], patch_size=patch_size,
        )

        # CFG
        v_cfg = v_uncond + guidance_scale * (v_cond - v_uncond)

        # Scheduler step (Flux2 VAE: no shift/scale)
        z = scheduler.step(v_cfg, t, z).prev_sample

    # VAE decode (denormalize → unpatchify → decode)
    return decode_vae_image(vae, z)


def validate(
    val_metadata_path: str,
    model_name: str = "black-forest-labs/FLUX.2-klein-base-9B",
    transformer_lora_path: str = None,
    qwen_lora_path: str = None,
    output_dir: str = "outputs/eval",
    device: str = "cuda",
    guidance_scale: float = 3.5,
    num_inference_steps: int = 28,
):
    from .dataset import DehazeValDataset
    from transformers import Qwen3ForCausalLM, Qwen2TokenizerFast

    val_dataset = DehazeValDataset(val_metadata_path)
    if len(val_dataset) == 0:
        raise ValueError(f"Validation dataset is empty: {val_metadata_path}")
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)

    vae, transformer, scheduler = load_inference_models(
        model_name, transformer_lora_path, qwen_lora_path, device,
    )

    text_encoder = Qwen3ForCausalLM.from_pretrained(
        model_name, subfolder="text_encoder", torch_dtype=torch.bfloat16,
    ).to(device)
    text_encoder.eval()

    if qwen_lora_path:
        text_encoder = PeftModel.from_pretrained(text_encoder, qwen_lora_path)
        text_encoder.eval()
        text_encoder.requires_grad_(False)

    tokenizer = Qwen2TokenizerFast.from_pretrained(
        model_name, subfolder="tokenizer",
    )

    os.makedirs(output_dir, exist_ok=True)
    psnr_list, ssim_list = [], []

    for batch in tqdm(val_loader, desc="Validating"):
        hazy = batch["hazy"]
        gt = batch["gt"]
        caption = (
            batch["caption"][0]
            if isinstance(batch["caption"], list)
            else batch["caption"]
        )

        clear = dehaze_single(
            vae=vae,
            transformer=transformer,
            scheduler=scheduler,
            text_encoder=text_encoder,
            tokenizer=tokenizer,
            hazy_image=hazy,
            prompt=caption,
            guidance_scale=guidance_scale,
            num_inference_steps=num_inference_steps,
            device=device,
        )

        clear_np = clear.squeeze(0).permute(1, 2, 0).cpu().float().clamp(0, 1).numpy()
        gt_np = gt.squeeze(0).permute(1, 2, 0).numpy()

        psnr_val = psnr(gt_np, clear_np, data_range=1.0)
        ssim_val = ssim(gt_np, clear_np, data_range=1.0, channel_axis=2)
        psnr_list.append(psnr_val)
        ssim_list.append(ssim_val)

    results = {
        "mean_psnr": float(np.mean(psnr_list)),
        "mean_ssim": float(np.mean(ssim_list)),
    }
    with open(Path(output_dir) / "metrics.json", "w") as f:
        json.dump(results, f, indent=2)

    print(
        f"Mean PSNR: {results['mean_psnr']:.2f} dB | "
        f"Mean SSIM: {results['mean_ssim']:.4f}"
    )
    return results
